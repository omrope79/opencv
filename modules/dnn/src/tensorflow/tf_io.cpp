// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of various functions which are related to Tensorflow models reading.
*/

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <fstream>
#include <vector>

#include "tf_io.hpp"
#include <unistd.h>
#include <fcntl.h>

namespace cv {
namespace dnn {

using std::string;
using std::map;
using namespace tensorflow;
using namespace ::google::protobuf;
using namespace ::google::protobuf::io;

static bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
#if defined(_WIN32)
    int fd = open(filename, O_RDONLY | O_BINARY);
#else
    int fd = open(filename, O_RDONLY);
#endif
    if (fd < 0) return false;

    FileInputStream* input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(input);
    
    // Fix: SetTotalBytesLimit only takes 1 argument in this version of Protobuf
    coded_input->SetTotalBytesLimit(INT_MAX);

    bool success = proto->ParseFromCodedStream(coded_input);

    delete coded_input;
    delete input;
    close(fd);
    return success;
}

static bool ReadProtoFromTextFile(const char* filename, Message* proto) {
#if defined(_WIN32)
    int fd = open(filename, O_RDONLY | O_BINARY);
#else
    int fd = open(filename, O_RDONLY);
#endif
    if (fd < 0) return false;

    FileInputStream* input = new FileInputStream(fd);
    bool success = TextFormat::Parse(input, proto);

    delete input;
    close(fd);
    return success;
}

// Implement the "OrDie" functions required by the header
void ReadTFNetParamsFromBinaryFileOrDie(const char* param_file, tensorflow::GraphDef* param) {
    if (!ReadProtoFromBinaryFile(param_file, param)) {
        CV_Error(Error::StsError, "Failed to parse GraphDef file: " + String(param_file));
    }
}

void ReadTFNetParamsFromTextFileOrDie(const char* param_file, tensorflow::GraphDef* param) {
    if (!ReadProtoFromTextFile(param_file, param)) {
        CV_Error(Error::StsError, "Failed to parse GraphDef text file: " + String(param_file));
    }
}

void ReadTFNetParamsFromBinaryBufferOrDie(const char* data, size_t len, tensorflow::GraphDef* param) {
    ArrayInputStream input(data, len);
    CodedInputStream coded_input(&input);
    
    // Fix: SetTotalBytesLimit only takes 1 argument
    coded_input.SetTotalBytesLimit(INT_MAX);
    
    if (!param->ParseFromCodedStream(&coded_input)) {
        CV_Error(Error::StsError, "Failed to parse GraphDef from binary buffer");
    }
}

void ReadTFNetParamsFromTextBufferOrDie(const char* data, size_t len, tensorflow::GraphDef* param) {
    ArrayInputStream input(data, len);
    if (!TextFormat::Parse(&input, param)) {
        CV_Error(Error::StsError, "Failed to parse GraphDef from text buffer");
    }
}

}
}
#endif
