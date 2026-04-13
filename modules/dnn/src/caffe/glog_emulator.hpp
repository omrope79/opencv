// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_GLOG_EMULATOR_HPP
#define OPENCV_DNN_GLOG_EMULATOR_HPP

#include <opencv2/core/base.hpp>

#include <sstream>
#include <string>

namespace cv {
namespace dnn {
namespace detail {

enum LogSeverity
{
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    FATAL = 3
};

class CheckStream
{
public:
    CheckStream(bool failed, const char* expr, const char* file, int line)
        : failed_(failed)
    {
        if (failed_)
            stream_ << file << ":" << line << ": Check failed: " << expr;
    }

    ~CheckStream()
    {
        if (failed_)
        {
            const std::string msg = stream_.str();
            CV_Error_(cv::Error::StsError, ("%s", msg.c_str()));
        }
    }

    template <typename T>
    CheckStream& operator<<(const T& value)
    {
        if (failed_)
            stream_ << value;
        return *this;
    }

    CheckStream& operator<<(std::ostream& (*manip)(std::ostream&))
    {
        if (failed_)
            manip(stream_);
        return *this;
    }

    CheckStream& operator<<(std::ios_base& (*manip)(std::ios_base&))
    {
        if (failed_)
            manip(stream_);
        return *this;
    }

private:
    bool failed_;
    std::ostringstream stream_;
};

class LogStream
{
public:
    LogStream(LogSeverity severity, const char* file, int line)
        : severity_(severity)
    {
        stream_ << file << ":" << line << ": ";
    }

    ~LogStream()
    {
        if (severity_ == FATAL)
        {
            const std::string msg = stream_.str();
            CV_Error_(cv::Error::StsError, ("%s", msg.c_str()));
        }
    }

    template <typename T>
    LogStream& operator<<(const T& value)
    {
        stream_ << value;
        return *this;
    }

    LogStream& operator<<(std::ostream& (*manip)(std::ostream&))
    {
        manip(stream_);
        return *this;
    }

    LogStream& operator<<(std::ios_base& (*manip)(std::ios_base&))
    {
        manip(stream_);
        return *this;
    }

private:
    LogSeverity severity_;
    std::ostringstream stream_;
};

}  // namespace detail
}  // namespace dnn
}  // namespace cv

#define LOG(severity) \
    cv::dnn::detail::LogStream(cv::dnn::detail::severity, __FILE__, __LINE__)

#define CHECK(condition) \
    cv::dnn::detail::CheckStream(!(condition), #condition, __FILE__, __LINE__)

#define CHECK_OP(op, val1, val2) \
    cv::dnn::detail::CheckStream(!((val1) op (val2)), #val1 " " #op " " #val2, __FILE__, __LINE__) \
        << " (" << (val1) << " vs " << (val2) << ")"

#define CHECK_EQ(val1, val2) CHECK_OP(==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(!=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(<, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(<=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(>, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(>=, val1, val2)

#endif  // OPENCV_DNN_GLOG_EMULATOR_HPP
